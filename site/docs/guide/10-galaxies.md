# 10. Galaxies, Sersic profiles, and velocity dispersion

Every lens in this repository is a galaxy first and a set of equations second.
Before the mass model, the sampler, and the evidence integral, there is a
physical object: a massive elliptical galaxy, sitting close enough to the line
of sight to a background source that its gravity bends the source's light into
arcs and rings. This chapter is about that object — what kind of galaxy it is,
how its light is described (the Sersic profile, and the ugly-looking constant
$b_n$ that makes it work), and how its stars' motion (the velocity dispersion
$\sigma_v$) predicts the one number every later chapter cares about, the
Einstein radius. It ends with a genuine physics puzzle: real massive
ellipticals turn out to have a total mass profile close to *isothermal*
($\gamma = 2$), for reasons no one fully derives from first principles. That
puzzle is the reason this repository's own prior on $\gamma$ is centered
exactly there — and the reason the money number, $\gamma = 1.103$, is worth
being suspicious of.

!!! abstract "What you can skip"
    You own standard deviation, exponentials, and simple ordinary differential
    equations already — skip past the notation, not the physics. If you
    already know what a de Vaucouleurs profile is and why $b_n$ exists, skip
    to [Velocity dispersion](#velocity-dispersion). If you already know that
    "isothermal" in "singular isothermal sphere" refers to a constant velocity
    dispersion, skip straight to [The isothermal conspiracy](#the-isothermal-conspiracy).

## Ellipticals { #ellipticals }

Galaxies split, broadly, into two dynamical families. Disk galaxies —
spirals, most star-forming galaxies — are supported against their own gravity
by *ordered rotation*: pick a star, and it is on a roughly circular orbit,
moving in the same sense as its neighbors. Elliptical galaxies are supported by
the opposite thing: *disordered* motion. Pick a star in an elliptical and it is
on some eccentric, randomly oriented orbit; there is no coherent spin field,
only a cloud of individual trajectories pointed every which way, the stellar
equivalent of a hot gas. The light is correspondingly featureless — smooth,
centrally concentrated, no spiral arms, no ongoing star formation, an old and
red stellar population — because there is no organizing structure left to draw
a pattern with.

Ellipticals are also, characteristically, the *most massive* galaxies in their
environment: the deepest potential wells around. That is not a coincidence of
which galaxies happen to appear in this repository's data — it is close to a
selection effect. Bending light by a detectable amount over an arcsecond-scale
region takes a deep, compact potential well, and ellipticals are the class of
galaxy built to have one. When Cikota et al. describe the lens in the Einstein
cross this repo reproduces, they call it plainly "a massive elliptical lens
galaxy (L1)" (`reproductions/cikota-2023/papers/main.tex:69`) — and the same
description fits essentially every lens galaxy this repository's campaigns
touch.

To subtract that galaxy's own light from an image before you can see the
arcs behind it (or to model it directly, as this repo's forward model does —
[Ch. 22](22-inference.md#the-forward-model)), you need a compact functional
form for how its surface brightness falls off with radius. Astronomers reach
for one function almost universally: the Sersic profile.

## The Sersic profile { #the-sersic-profile }

Surface brightness $I(R)$ — light per unit solid angle at projected radius $R$
([Ch. 9](09-units.md#surface-brightness)) — for a Sersic profile is

$$
I(R) = I_e \exp\left\{-b_n\left[\left(\frac{R}{R_e}\right)^{1/n} - 1\right]\right\}.
$$

Three quantities carry the physics. $R_e$, the *effective radius*, is defined
so that exactly half the galaxy's total light comes from $R < R_e$ — it is a
half-light radius by construction, not a fitted scale length. $I_e = I(R_e)$
is the surface brightness there. $n$, the *Sersic index*, is a shape
parameter: $n=1$ recovers the exponential profile of a disk galaxy; $n=4$
recovers de Vaucouleurs's 1948 law for elliptical light,
$I \propto \exp\{-b(R/R_e)^{1/4}\}$; other values of $n$ interpolate and
extrapolate between a shallow, extended profile and a sharply cuspy one.

$b_n$ is the piece that looks like an arbitrary fitting constant and is not.
Because $R_e$ *must* enclose exactly half the light regardless of $n$, $b_n$ is
not a free knob — once you fix $n$, $b_n$ is *determined* by that requirement.
Write the total luminosity as an integral over the whole plane,

$$
L = \int_0^\infty I(R)\, 2\pi R\, dR,
$$

and substitute $x = b_n (R/R_e)^{1/n}$, so $R = R_e (x/b_n)^n$. The integral
turns into a Gamma-function form,
$L = 2\pi n R_e^2 I_e\, e^{b_n} b_n^{-2n}\, \Gamma(2n)$, and demanding that
the light within $R_e$ equal exactly half of $L$ becomes a condition on the
*regularized incomplete* Gamma function,

$$
P(2n,\, b_n) = \frac{1}{2}.
$$

This equation has no closed form in elementary functions — $b_n$ is defined
implicitly, one value per $n$, and has to be solved numerically. What
everyone actually uses, including this repository's own gigalens vendor code
(`reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/light/sersic.py:16`)
and this guide's own `site/guide_src/lensing.py:157`, is the
Capaccioli/Ciotti–Bertin fitting formula, accurate to well under a percent for
$1 \lesssim n \lesssim 10$ (it degrades fast below that — at $n=0.5$ it is
already off by nearly 3%, [Exercise 10.1](#exercises) quantifies the $n=1$ end
of that range):

$$
b_n \approx 1.9992\,n - 0.3271.
$$

At $n=1$ (exponential disk) this gives $b_n = 1.6721$
<!-- check: ch10.bn_n1 = 1.6721 ± 0.0001 -->; at $n=4$ (de Vaucouleurs) it
gives $b_n = 7.6697$ <!-- check: ch10.bn_n4 = 7.6697 ± 0.0001 -->. The
increase is a factor of about four and a half
<!-- check: ch10.bn_ratio_n4_over_n1 = 4.587 ± 0.001 -->, which is the
normalization constant's way of saying that a de Vaucouleurs profile is far
more centrally cuspy — most of the light is crammed close to $R_e$ from
outside, with a long faint outer wing — than a disk's gentler exponential
fall-off.

<figure markdown="span">
  ![Sersic surface-brightness profiles for four values of the Sersic index n](figures/ch10-sersic-profiles-light.svg#only-light){ width="90%" }
  ![Sersic surface-brightness profiles for four values of the Sersic index n](figures/ch10-sersic-profiles-dark.svg#only-dark){ width="90%" }
  <figcaption markdown="span">**Figure 10.1.** $I(R)/I_e$ for $n = 0.5, 1, 2, 4$,
  each rescaled by its own $b_n$. Every curve, regardless of $n$, passes
  through 1 at $R/R_e = 1$ — that is what solving $P(2n, b_n) = 1/2$ for $b_n$
  *buys you*: a single normalization convention under which $R_e$ means the
  same thing (half the light) no matter how steep or shallow the profile is.</figcaption>
</figure>

The campaign's own lens-light model does not commit to a single $n$ up front.
It fits *four* separate Sersic components for the lens galaxy's light (plus a
Sersic-and-shapelet source), each drawing its Sersic index from a
$\mathrm{Uniform}(0.5, 8.0)$ prior
(`reproductions/claude-giga-lens/cgl/e2.py:101`) — spanning everything from a
disky $n=0.5$ profile to a super-de-Vaucouleurs $n=8$ cusp, because a real
galaxy's light does not arrive pre-labeled with its own index. [Ch. 22](22-inference.md#the-forward-model)
covers the full four-component model this feeds into.

## Velocity dispersion { #velocity-dispersion }

!!! tip "You already know this"
    Velocity dispersion is exactly a standard deviation: the same statistical
    object as the $\sigma$ in a Gaussian likelihood
    ([Ch. 8](08-probability.md#chi-squared)) or the noise term in any
    regression loss. A galaxy with $\sigma_v = 300$ km/s has stars whose
    line-of-sight velocities are spread three times as wide as one with
    $\sigma_v = 100$ km/s. Nothing about the astrophysics changes what kind of
    object $\sigma_v$ is.

Because an elliptical is not rotating in any organized way, you cannot read
its mass off a rotation curve the way you can for a disk. What you *can*
measure — from the width of an absorption line in its spectrum, the subject of
[Ch. 12](12-spectroscopy.md#sigma-v-from-lines) — is $\sigma_v$, the spread of
its stars' line-of-sight velocities. The question this section answers is why
that single number is enough to predict an Einstein radius.

Start from hydrostatic equilibrium for a self-gravitating,
velocity-dispersion-supported sphere — astronomers call this the (isotropic)
Jeans equation; it is the exact mechanical analogue of $dP/dr = -\rho\,GM(r)/r^2$
for a gas, with pressure $P$ replaced by $\rho\sigma_v^2$:

$$
\frac{d}{dr}\big(\rho(r)\,\sigma_v^2\big) = -\rho(r)\,\frac{GM(r)}{r^2},
\qquad M(r) = 4\pi\int_0^r \rho(r')\,r'^2\,dr'.
$$

"Isothermal" means $\sigma_v$ does not depend on $r$ — literally the
mechanical statement of a system held at constant temperature, since for an
ideal gas $P = \rho\,k_BT/m$ and $\sigma_v^2$ plays the role $k_BT/m$ plays
there. With $\sigma_v$ pulled outside the derivative and no other length scale
anywhere in the problem, dimensional analysis alone forces $\rho(r)$ to be a
pure power law; try $\rho(r) = A/r^2$ and check that it is self-consistent.
Then $M(r) = 4\pi A r$ (density $\times$ area integrates linearly in $r$), so

$$
\text{LHS} = \sigma_v^2\frac{d}{dr}\!\left(\frac{A}{r^2}\right) = -\frac{2A\sigma_v^2}{r^3},
\qquad
\text{RHS} = -\frac{A}{r^2}\cdot\frac{G\cdot 4\pi A r}{r^2} = -\frac{4\pi GA^2}{r^3}.
$$

Both sides scale as $r^{-3}$ — the ansatz was consistent — and equating the
coefficients gives $A = \sigma_v^2/(2\pi G)$. So a constant-$\sigma_v$
("isothermal") sphere has density

$$
\rho(r) = \frac{\sigma_v^2}{2\pi G\,r^2}.
$$

That is exactly $\rho \sim r^{-\gamma}$ with $\gamma = 2$: the isothermal
sphere *is* the $\gamma=2$ profile, and this derivation is where the word
"isothermal" earns its meaning rather than being a label attached to a
formula. It also explains the other half of "singular isothermal sphere": the
Abel projection of [Ch. 3](03-integrals.md#the-abel-projection) turns
$\rho \sim r^{-\gamma}$ into a surface density $\Sigma \sim R^{1-\gamma}$, so
at $\gamma=2$, $\Sigma(R) \propto 1/R$ — finite everywhere except a genuine
divergence at $R=0$. That divergence is the "singular."

The same profile also gives you a disk galaxy's flat rotation curve for free:
the circular velocity satisfies
$v_{\mathrm{circ}}^2(r) = GM(r)/r = 4\pi GA = 2\sigma_v^2$ — constant in $r$,
independent of $A$'s value. A flat rotation curve (rotation-supported
systems) and a constant velocity dispersion (pressure-supported systems) are
the *same* statement, $\rho \sim r^{-2}$, observed through two different
kinematic windows.

[Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) takes this density
profile the rest of the way, combining it with the definition of $\theta_{\mathrm{E}}$
as the radius where the mean interior convergence equals exactly 1 to arrive
at

$$
\theta_{\mathrm{E}} = 4\pi\left(\frac{\sigma_v}{c}\right)^{2}\frac{D_{\mathrm{ds}}}{D_{\mathrm{s}}}
$$

(Hsu+2025 Eq. 1, implemented at `site/guide_src/cosmo.py:79` and
`site/guide_src/lensing.py:170`; recall
$D_{\mathrm{ds}} \neq D_{\mathrm{s}} - D_{\mathrm{d}}$,
[Ch. 15](15-distances.md#distances-do-not-add)). You can already use the
formula here, since its only missing ingredient beyond what you just derived
is that one projection step. For a fiducial massive elliptical —
$\sigma_v = 250$ km/s at $z_l=0.5$, $z_s=2.0$, the same system
[Ch. 19](19-einstein-radius.md) uses — it gives
$\theta_{\mathrm{E}} = 1.145''$
<!-- check: ch10.theta_e_typical_elliptical = 1.145 ± 0.01 -->. Because the
formula is quadratic in $\sigma_v$, doubling the velocity dispersion — from
150 km/s to 300 km/s — exactly quadruples the Einstein radius, from
$0.412''$ <!-- check: ch10.theta_e_150 = 0.412 ± 0.005 --> to
$1.649''$ <!-- check: ch10.theta_e_300 = 1.649 ± 0.01 -->, a ratio of $4.0$
<!-- check: ch10.theta_e_ratio_2x_sigma = 4.0 ± 1e-6 -->. That steep scaling
is also why lensing is rare: only fairly massive ellipticals, with $\sigma_v$
comfortably above $\sim 150$ km/s, produce an Einstein radius large enough to
resolve against a survey's seeing at all — the same resolution constraint
[Ch. 27](27-discovery.md#deriving-the-wall) derives from the pixel scale
directly.

## The isothermal conspiracy { #the-isothermal-conspiracy }

Here is the puzzle the derivation above sets up but does not resolve. A real
massive elliptical's gravity is built from two physically unrelated
ingredients. The *stars* follow something like the Sersic profile just
derived — not a power law at all, an exponential of a fractional power,
shallow near the center and falling off far faster than $r^{-2}$ at large
radius. The *dark matter* — inferred, historically, from exactly the flat
rotation curves derived above staying flat well past the edge of a disk
galaxy's visible light, which under Newtonian gravity requires unseen mass out
there — is typically modeled with an NFW profile, whose logarithmic slope is
not constant either: shallower than isothermal near the center, steeper than
isothermal far outside.

Neither component, on its own, is isothermal. And yet decades of joint
lensing-plus-dynamics measurements of real massive ellipticals find that the
*total* mass profile — stars and dark matter summed, which is all strong
lensing and stellar dynamics ever weigh — comes out close to isothermal
($\gamma \approx 2$) over the radii those measurements actually probe, roughly
the effective radius $R_e$ this chapter's Sersic profile defines. Two
physically unrelated components, with unrelated profile shapes, adding up to
something close to a single clean power law is not a prediction of any
first-principles argument. It is an empirical regularity astronomers have
been trying to explain by appeal to galaxy-formation physics — feedback,
adiabatic contraction — for decades, with no fully settled account. Some of
the literature calls it the bulge–halo conspiracy; this book, following the
outline's own name for it, calls it the isothermal conspiracy.

It is not a throwaway fact for this repository. It is written directly into
the campaign's own mass model: the EPL slope prior
(`reproductions/claude-giga-lens/cgl/e2.py:110`) is
$\gamma \sim \mathrm{TruncatedNormal}(2.0,\ 0.25,\ \mathrm{low}=1.0,\ \mathrm{high}=2.7)$
<!-- check: ch10.gamma_prior_mean = 2.0 ± 1e-9 -->
<!-- check: ch10.gamma_prior_sigma = 0.25 ± 1e-9 -->
— centered exactly on the conspiracy, with a width reflecting the real
galaxy-to-galaxy scatter astronomers measure around it, and hard walls at 1.0
and 2.7 because slopes that shallow or that steep essentially never occur in
real massive ellipticals. That prior is not a mathematical convenience chosen
for this campaign. It is this section, compiled.

Hold the number 2.0 in mind. This repository's own headline measurement for
one galaxy, $\gamma_{\mathrm{binned}}(\mathrm{corr}, \mathrm{low}) = 1.103$
<!-- check: ch25.gamma_money = 1.103 ± 0.008 -->, sits $3.588$
<!-- check: ch25.prior_sigmas_below_prior_mean = 3.588 ± 0.001 -->
prior-sigma below that center — closer to the prior's own hard floor at 1.0
than to its mean. For comparison, the value [Ch. 25](25-money-number.md)
treats as the *trustworthy* anchor, $\gamma = 1.433$, sits only $2.268$
<!-- check: ch10.anchor_prior_sigmas = 2.268 ± 0.001 --> prior-sigma below
2.0 — still a real departure from isothermal, but a smaller one. Whether
1.103's larger departure reflects a genuinely unusual galaxy or a mismodeled
noise covariance is exactly what Chapters 20 through 26 exist to adjudicate.

!!! note "γ Ledger"
    **What this chapter rules in or out about $\gamma = 1.103$:** nothing,
    yet — this chapter fixes the *expectation*, not the *verdict*. Real
    massive ellipticals cluster close to $\gamma = 2$ for real, if
    incompletely understood, physical reasons; that is why this repository's
    own prior is centered there rather than at some arbitrary number. Against
    that expectation, $\gamma = 1.103$ is a large-looking departure — 3.588
    prior-sigma — but a large prior-sigma is a statement about the *prior*,
    not about the *data*. Whether the data actually demand a galaxy this far
    from isothermal is [Ch. 25](25-money-number.md#the-sigma-arithmetic)'s
    question, and whether that demand survives a saddle-point posterior is
    [Ch. 26](26-the-saddle.md)'s.

## Connect to the repo { #connect }

- `site/guide_src/lensing.py:157` (`sersic_bn`) and `:162` (`sersic`) — the
  Capaccioli fit, matching the campaign's production code exactly.
- [`reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/light/sersic.py:16`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/light/sersic.py#L16) —
  gigalens' own `bn = 1.9992 * n_sersic - 0.3271`.
- [`reproductions/claude-giga-lens/cgl/e2.py:100`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L100)–`123` —
  the four lens-light Sersic priors and the source's; line `101`
  ($n \sim \mathrm{Uniform}(0.5,8.0)$) and line `110`
  ($\gamma \sim \mathrm{TruncatedNormal}(2.0,0.25,1.0,2.7)$), the isothermal
  conspiracy written as code.
- `site/guide_src/cosmo.py:79` (`theta_e_from_sigma_v`) and
  `site/guide_src/lensing.py:170` (`theta_e_sis`) — the SIS
  $\theta_{\mathrm{E}}$–$\sigma_v$ relation this chapter derives the origin of
  and [Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v) applies.
- [`reproductions/cikota-2023/papers/main.tex:69`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/cikota-2023/papers/main.tex#L69) —
  "a massive elliptical lens galaxy (L1)," the description that fits nearly
  every lens in this repository's campaigns.

## Exercises { #exercises }

??? question "Exercise 10.1 — b_n exactly, versus the fit"
    $b_n$ is defined by $P(2n, b_n) = 1/2$, where $P$ is the regularized lower
    incomplete Gamma function. In Python, `scipy.special.gammaincinv(a, p)`
    solves exactly this: it returns the $x$ such that $P(a, x) = p$. Compute
    the *exact* $b_n$ for $n=1$ and $n=4$ via `gammaincinv(2*n, 0.5)`, compare
    to the Capaccioli fit's $1.6721$ and $7.6697$, and explain — in terms of
    what the fit was tuned for — why one comparison is far tighter than the
    other.

    ??? success "Solution"
        ```python
        from scipy.special import gammaincinv
        gammaincinv(2*1, 0.5)   # 1.6783
        gammaincinv(2*4, 0.5)   # 7.6692
        ```
        At $n=4$ the fit ($7.6697$) and the exact value
        ($7.6692$ <!-- check: ch10.bn_n4_exact = 7.6692 ± 0.0002 -->) agree to
        four significant figures — a relative error under $0.01\%$. At $n=1$
        the fit ($1.6721$) misses the exact value
        ($1.6783$ <!-- check: ch10.bn_n1_exact = 1.6783 ± 0.0002 -->) by about
        $0.4\%$, two orders of magnitude worse. The Capaccioli/Ciotti–Bertin
        formula was fit to reproduce de Vaucouleurs's $n=4$ profile — the one
        astronomers actually cared about getting right for elliptical
        galaxies — and is a linear approximation to a genuinely nonlinear
        function everywhere else. It is *good enough* across the whole range
        this repo's own $n \sim \mathrm{Uniform}(0.5, 8.0)$ prior spans, but
        "good enough" is not "exact," and $n=1$ is where that shows up first.

??? question "Exercise 10.2 — the flat rotation curve, from the same profile"
    Using $\rho(r) = \sigma_v^2/(2\pi G r^2)$ and
    $M(r) = 4\pi\int_0^r \rho(r')r'^2\,dr'$, show that the circular velocity
    $v_{\mathrm{circ}}(r) = \sqrt{GM(r)/r}$ is independent of $r$, and find
    it in terms of $\sigma_v$.

    ??? success "Solution"
        $M(r) = 4\pi\int_0^r \dfrac{\sigma_v^2}{2\pi G r'^2}\,r'^2\,dr' = \dfrac{2\sigma_v^2}{G}\,r$
        — the $r'^2$ from the volume element exactly cancels the
        $r'^{-2}$ in $\rho$, so the integrand is constant and $M(r)$ is
        linear in $r$. Then

        $$
        v_{\mathrm{circ}}^2(r) = \frac{GM(r)}{r} = \frac{G}{r}\cdot\frac{2\sigma_v^2}{G}r = 2\sigma_v^2,
        $$

        with no $r$ left anywhere — flat, at
        $v_{\mathrm{circ}} = \sqrt{2}\,\sigma_v$. This is the same
        cancellation, for the same reason, that made the density ansatz
        self-consistent in the main text: an $r^{-2}$ density is precisely
        the one power law whose
        enclosed mass grows linearly, which is precisely what a flat rotation
        curve (or a constant velocity dispersion) requires.

??? question "Exercise 10.3 — doubling sigma_v"
    Confirm, using `cosmo.theta_e_from_sigma_v` (`site/guide_src/cosmo.py:79`),
    that doubling $\sigma_v$ at fixed $z_l, z_s$ exactly quadruples
    $\theta_{\mathrm{E}}$, and say in one sentence why the exponent is 2 and
    not, say, 1.

    ??? success "Solution"
        ```python
        import cosmo
        cosmo.theta_e_from_sigma_v(150.0, 0.5, 2.0)   # 0.412"
        cosmo.theta_e_from_sigma_v(300.0, 0.5, 2.0)   # 1.649"
        # ratio: 4.0
        ```
        That ratio is exactly the $4.0$ from the main text
        <!-- check: ch10.theta_e_ratio_2x_sigma = 4.0 ± 1e-6 -->. The formula
        $\theta_{\mathrm{E}} = 4\pi(\sigma_v/c)^2 D_{\mathrm{ds}}/D_{\mathrm{s}}$
        is quadratic in $\sigma_v$ because $\theta_{\mathrm{E}}$ traces back
        to $\rho(r) \propto \sigma_v^2/r^2$ — the $\sigma_v^2$ sitting in the
        density's normalization propagates linearly through $M(r)$, through
        $\Sigma(R)$, and through the mean-convergence definition of
        $\theta_{\mathrm{E}}$, none of which introduces another power of
        $\sigma_v$ or cancels the one that is there.

??? question "Exercise 10.4 — how surprised should the anchor make you?"
    The campaign's $\gamma$ prior is centered at 2.0 with $\sigma = 0.25$
    (`reproductions/claude-giga-lens/cgl/e2.py:110`). [Ch. 25](25-money-number.md)
    treats $\gamma = 1.433$ as the *anchor* — the value it is willing to trust.
    How many prior-sigma is the anchor from isothermal, and how does that
    compare to the money number's own $3.588$?

    ??? success "Solution"

        $$
        \frac{2.0 - 1.433}{0.25} = 2.268
        $$

        <!-- check: ch10.anchor_prior_sigmas = 2.268 ± 0.001 -->
        The anchor is already a real departure from isothermal — 2.268
        prior-sigma is not nothing — but it is noticeably smaller than the
        money number's 3.588. Both numbers describe *surprise relative to a
        prior expectation*, not evidence from the data; neither, by itself,
        tells you which measurement to believe. That is a different question,
        with a different kind of arithmetic, and it is exactly what
        [Ch. 25](25-money-number.md#the-sigma-arithmetic) works through.
