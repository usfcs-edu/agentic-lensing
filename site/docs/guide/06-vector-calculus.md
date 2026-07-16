# Divergence, Laplacian, Green's functions: the potential trio

This chapter derives one equation and one convolution, and everything from
Ch. 16 onward cashes them in without re-deriving them. The equation is
$\nabla^2\psi = 2\kappa$, alongside $\boldsymbol\alpha = \nabla\psi$: the
deflection field is the gradient of a scalar potential, and the convergence
(the mass, in the units Ch. 15 built) is half that potential's Laplacian. The
convolution is the reason every lensing potential you will meet — point mass,
SIS, SIE, EPL — is built from a logarithm: the logarithm is the Green's
function of the 2-D Laplacian, the same object that turns "some mass
distribution" into "the potential it sources." Divergence and the Laplacian
are the last two vector-calculus tools Part IV needs; this chapter builds them
from the gradient and Jacobian you already have (Ch. 4) and hands you back the
single identity that the rest of the book calls "the trio."

!!! abstract "What you can skip"
    If you remember the divergence theorem (flux through a closed surface
    equals the integral of divergence over the enclosed volume) from an
    undergraduate course, skip straight to [Green's function of the 2-D
    Laplacian](#greens-function) — the general theory isn't re-derived here,
    only the one 2-D consequence this repo needs. If you don't remember it, or
    never proved it, the short argument in the next section is the whole
    proof you need for this book; it is not restated in later chapters.

## Divergence and the Laplacian { #divergence-laplacian }

For a vector field $\mathbf{F}(x,y) = (F_x, F_y)$, the **divergence** is the
scalar

$$
\nabla\cdot\mathbf{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y}.
$$

Read it as a flux density: $\nabla\cdot\mathbf{F}$ at a point measures how
much $\mathbf{F}$ is "spreading out" of an infinitesimal neighborhood of that
point, per unit area. The divergence theorem makes this precise:

$$
\oint_{\partial\Omega} \mathbf{F}\cdot\hat n \, d\ell = \int_\Omega \nabla\cdot\mathbf{F} \, dA
$$

for any region $\Omega$ — but the intuition ("divergence = local source
density") is what you need going forward.

The **Laplacian** of a scalar field $f$ is the divergence of its gradient:

$$
\nabla^2 f \;=\; \nabla\cdot(\nabla f) \;=\; \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}.
$$

You already built the machinery for this in [Ch. 4](04-multivariable.md#gradient-jacobian-hessian):
$\nabla^2 f$ is exactly the **trace** of the Hessian of $f$ — the sum of its
diagonal entries, and (per [Ch. 5](05-linear-algebra.md#symmetric-2x2)'s
eigendecomposition of a symmetric matrix) the sum of its eigenvalues too,
since trace is basis-independent. That single fact — Laplacian = trace of
Hessian — is the entire content of the derivation in [Poisson's equation for
lensing](#poisson-for-lensing) below.

A field with $\nabla^2 f = 0$ everywhere is called **harmonic**. Harmonic
functions have no local sources or sinks: the value at any point equals the
average of $f$ over any small circle centered there. That mean-value property
is why a harmonic potential can only vary because of what's happening
*somewhere else* — never because of anything at the point itself.

One more identity, used silently every time this repo differentiates a
deflection field instead of writing down a potential: for any smooth scalar
$f$, $\nabla\times(\nabla f) \equiv 0$ — the curl of a gradient is
identically zero, because mixed partial derivatives commute
($\partial_x\partial_y f = \partial_y\partial_x f$). A field that comes from a
potential can never rotate. This is why gigalens' EPL profile is allowed to
hand you $\boldsymbol\alpha(\theta_1,\theta_2)$ as a closed-form deflection
and never construct $\psi$ at all (`gigalens/jax/profiles/mass/epl.py:19`,
linked in [Connect to the repo](#connect)): as long as
$\partial\alpha_{\theta_1}/\partial\theta_2 = \partial\alpha_{\theta_2}/\partial\theta_1$
holds identically in the closed form, a $\psi$ is guaranteed to exist whose
gradient it is, even though nobody ever writes it down.

!!! tip "You already know this"
    A continuous normalizing flow (a Neural ODE, $dz/dt = f(z,t)$) evolves its
    log-density as $d(\log p)/dt = -\nabla_z \cdot f(z,t)$ — the divergence of
    the flow field, which is also the trace of its Jacobian. That is the
    continuous-time cousin of the $-\log|\det J|$ a discrete flow layer pays
    per step (already on this guide's Log-Det Ledger, opened in
    [Ch. 4](04-multivariable.md#the-log-det-ledger)). Divergence is "how much
    a vector field spreads a density out." In this chapter it spreads mass
    out of a region of sky; in a flow it spreads probability out of a region
    of latent space. Same operation, different density.

## Green's function of the 2-D Laplacian { #greens-function }

A **Green's function** is the impulse response of a linear operator: solve
$L[G] = \delta$ (a unit point source), and the solution for *any* source $s$
is the convolution $f = G * s$. For the 2-D Laplacian, the Green's function is
a logarithm, and you can derive it with nothing but the divergence theorem.

Put a unit point source at the origin. By symmetry, the field
$\mathbf{E} = \nabla G$ it sources must be radial: $\mathbf{E}(r) = E(r)\,\hat r$.
Apply the divergence theorem to a disk of radius $r$ centered on the source.
The enclosed source is $1$ regardless of $r$ (it's a point, and any disk
around it encloses all of it), so the flux out through the boundary circle
must also be $1$ for every $r > 0$:

$$
E(r)\cdot 2\pi r = 1 \quad\Longrightarrow\quad E(r) = \frac{1}{2\pi r}.
$$

Integrating $E(r) = dG/dr$ gives

$$
G(r) = \frac{1}{2\pi}\ln r + \text{const}. \label{eq:green-2d}
$$

This is the differential-equation route to the same fact
[Ch. 3](03-integrals.md#why-log-potential) derived by integrating Gauss's law
directly over a 2-D mass distribution: a 2-D point source has a *logarithmic*
potential, not the $1/r$ you'd expect from 3-D intuition. $\eqref{eq:green-2d}$
is harmonic everywhere except at $r=0$ — checkable numerically, since
"harmonic away from the source" is the testable half of "all the source sits
at one point."

**Worked check.** Take the lensing potential of a point mass,
$\psi_{\text{pt}}(\theta_1,\theta_2) = \theta_{\mathrm E}^2\ln r$ with
$\theta_{\mathrm E}=1''$ and $r=\sqrt{\theta_1^2+\theta_2^2}$ — the same
functional form as $\eqref{eq:green-2d}$, with the point mass folded into the
prefactor. Evaluate its Laplacian by central finite differences at
$(\theta_1,\theta_2) = (1.2'', 0.9'')$, i.e. $r_0 = 1.5''$
<!-- check: ch06.r0 = 1.5 ± 0.0001 -->. The five-point stencil returns
$-2.2\times10^{-8}$ <!-- check: ch06.point_mass_laplacian_offcenter = -2.2e-8 ± 1e-6 -->:
zero to floating-point precision, off by roughly $10^{-8}$ purely from
`h = 1e-4` truncation error — confirming that away from the point, this
potential carries no convergence at all. (This point-mass potential is *not*
one of this repo's mass profiles — its galaxies are extended isothermal-like
systems, Ch. 20's SIS/SIE/EPL family — but it is the cleanest illustration of
the Green's function, because it isolates the singular source term from
everything else.)

## Poisson's equation for lensing { #poisson-for-lensing }

Newtonian gravity obeys $\nabla^2\Phi = 4\pi G\rho$ — the same divergence-
theorem argument as the last section, sourced by mass instead of a unit point
charge. [Ch. 16](16-deflection.md#the-thin-lens) works out how projecting this
along the line of sight and rescaling by the distance factors and
$\Sigma_{\mathrm{cr}}$ ([Ch. 15](15-distances.md#sigma-crit)) collapses the
3-D equation to a 2-D one for the dimensionless lensing potential
$\psi(\boldsymbol\theta)$:

$$
\nabla^2\psi = 2\kappa, \qquad \boldsymbol\alpha = \nabla\psi. \label{eq:poisson-lensing}
$$

This is a *different* factor of two from [Ch. 16](16-deflection.md#the-factor-of-two)'s
GR-is-twice-Newton deflection — two unrelated 2's sitting in the same
subject, by coincidence of convention, not physics. Keep them apart.

You can derive exactly where $\eqref{eq:poisson-lensing}$'s "2" comes from
using only algebra you already have, with no new physics. Since
$\boldsymbol\alpha=\nabla\psi$, the Jacobian of the deflection field is the
Hessian of $\psi$: $\partial\boldsymbol\alpha/\partial\boldsymbol\theta = H_\psi$.
The lens Jacobian ([Ch. 17](17-lens-equation.md#the-lens-equation), defined by
$A = \partial\boldsymbol\beta/\partial\boldsymbol\theta$ with $\boldsymbol\beta = \boldsymbol\theta - \boldsymbol\alpha$)
is therefore $A = I - H_\psi$. The notation contract also fixes
$A = (1-\kappa)I - \Gamma$, with $\Gamma$ the *traceless* shear matrix.
Equating the two expressions for $A$:

$$
H_\psi = I - A = \kappa I + \Gamma.
$$

Take the trace of both sides. $\mathrm{tr}(\Gamma) = 0$ by construction (that
is what "traceless" means), and $\mathrm{tr}(H_\psi) = \nabla^2\psi$ by
definition from the previous section. So

$$
\nabla^2\psi = \mathrm{tr}(H_\psi) = 2\kappa + \mathrm{tr}(\Gamma) = 2\kappa.
$$

The "2" is not a physical constant to look up — it is the trace of an
identity matrix in 2-D, appearing because $\kappa$ was *defined* as the
isotropic (trace) part of $H_\psi$ and $\Gamma$ was defined to soak up
everything traceless. Equivalently, since $\boldsymbol\alpha=\nabla\psi$, you
can skip the Hessian and go straight for divergence: $\nabla\cdot\boldsymbol\alpha
= \nabla\cdot(\nabla\psi) = \nabla^2\psi = 2\kappa$.

**Worked check.** `site/guide_src/lensing.py`'s `L.sis_deflection` is the SIS
deflection $\boldsymbol\alpha = \theta_{\mathrm E}\,\hat\theta$
([Ch. 20](20-profiles.md#sis-to-sie) works out why an isothermal profile gives
a *constant-modulus* deflection). Its analytic convergence,
$\kappa_{\mathrm{SIS}}(\theta) = \theta_{\mathrm E}/(2\theta)$, is the same
form [Ch. 19](19-einstein-radius.md#the-mean-convergence-identity) uses as
`lambda t: 0.5 * 1.0 / t`. At $\theta = 1.5''$ with $\theta_{\mathrm E}=1''$:

$$
\kappa_{\mathrm{SIS}}(1.5) = \frac{1}{2\times 1.5} = \frac{1}{3} \approx 0.3333.
$$

<!-- check: ch06.sis_kappa_analytic = 0.3333 ± 0.0001 -->
So $2\kappa \approx 0.6667$ <!-- check: ch06.sis_two_kappa = 0.6667 ± 0.0001 -->.
Now compute $\nabla\cdot\boldsymbol\alpha$ with no shortcut: finite-difference
`L.sis_deflection` itself at four points around $(1.2'', 0.9'')$ and sum
$\partial\alpha_{\theta_1}/\partial\theta_1 + \partial\alpha_{\theta_2}/\partial\theta_2$.
The result is $0.6667$ <!-- check: ch06.sis_div_alpha_numeric = 0.6667 ± 0.0001 -->,
matching $2\kappa$ to a residual of about $2\times10^{-9}$
<!-- check: ch06.sis_poisson_residual = 0.0 ± 1e-6 --> — pure finite-difference
truncation, nothing else. This is not a new computation dressed up: it is
*exactly* what
`L.kappa_gamma_from_jacobian` already does at `site/guide_src/lensing.py:133`
(`kappa = 1.0 - 0.5 * (a11 + a22)`), which you met in
[Ch. 5](05-linear-algebra.md#symmetric-2x2) as an eigenvalue decomposition and
in [Ch. 18](18-magnification.md#magnification-is-a-jacobian) as a
magnification ingredient. That line differentiates $\boldsymbol\beta$, not
$\boldsymbol\alpha$, and gets $\kappa$ by $1 - \mathrm{tr}(A)/2$ rather than
$\mathrm{tr}(H_\psi)/2$ — but $\mathrm{tr}(A) = 2 - \mathrm{tr}(H_\psi)$, so
it is the identical arithmetic, one substitution away.

The synthesis of this chapter's two sections: $\kappa$ is the *source*,
$\psi$ is what the Green's function convolution of that source produces,

$$
\psi(\boldsymbol\theta) = \frac{1}{\pi}\int \kappa(\boldsymbol\theta')\,
\ln|\boldsymbol\theta-\boldsymbol\theta'| \, d^2\theta' + (\text{linear terms}),
$$

and $\boldsymbol\alpha=\nabla\psi$ differentiates that convolution. gigalens
never performs this integral — SIS, SIE and EPL all have closed-form
deflections precisely because someone already did the convolution by hand —
but the closed form is only trustworthy because $\eqref{eq:poisson-lensing}$
guarantees it corresponds to *some* $\kappa$ in the first place.

## Connect to the repo { #connect }

- [`site/guide_src/lensing.py:101`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L101)
  (`lens_jacobian`) and
  [`site/guide_src/lensing.py:133`](https://github.com/usfcs-edu/agentic-lensing/blob/main/site/guide_src/lensing.py#L133)
  (`kappa_gamma_from_jacobian`) are this chapter's identity, discretized: a
  finite-difference Hessian of the deflection field, traced and halved.
- [`reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/epl.py:19`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/vendor/gigalens-sean/src/gigalens/jax/profiles/mass/epl.py#L19)
  is the production EPL deflection: a closed-form $\boldsymbol\alpha$ that
  never materializes $\psi$, relying silently on curl-free-ness for its own
  consistency.
- The letter $A$ is doing two unrelated jobs in this repo: the lens Jacobian
  here, and the Gaussian-normal matrix in
  [`reproductions/claude-giga-lens/cgl/marg.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/marg.py)'s
  Occam term ([Ch. 22](22-inference.md#the-occam-term)). Same symbol,
  different matrix — the notation contract flags this collision on purpose.

## Exercises { #exercises }

??? question "Exercise 6.1 — Trace is basis-independent, so $\kappa$ doesn't care which way you point your axes"
    Let $R$ be a 2-D rotation matrix. Show that
    $\mathrm{tr}(R^\top H_\psi R) = \mathrm{tr}(H_\psi)$ for any matrix
    $H_\psi$. What does this tell you about computing $\kappa$ in a rotated
    coordinate system — say, aligned with a galaxy's major axis instead of
    RA/Dec?

    ??? success "Solution"
        Trace is invariant under conjugation:
        $\mathrm{tr}(R^\top H_\psi R) = \mathrm{tr}(H_\psi R R^\top) = \mathrm{tr}(H_\psi I) = \mathrm{tr}(H_\psi)$,
        using the cyclic property of trace and $RR^\top = I$ for a rotation.
        Since $\nabla^2\psi = \mathrm{tr}(H_\psi)$, the Laplacian — and hence
        $\kappa$ — is the same number no matter which orthonormal axes you
        differentiate in. This is why $\kappa$ can be quoted as a single
        scalar per point without ever specifying an orientation, while
        $\gamma_1,\gamma_2$ (the *traceless* part) rotate into each other and
        must always be quoted with a position angle — exactly the $q,\phi$
        machinery of [Ch. 20](20-profiles.md#ellipticity).

??? question "Exercise 6.2 — The Green's function's flux doesn't care about the radius either"
    Using $E(r) = 1/(2\pi r)$ from $\eqref{eq:green-2d}$, show by direct
    computation that the flux $\oint \mathbf{E}\cdot\hat n\,d\ell$ through a
    circle of radius $R$ is exactly $1$ for *every* $R>0$, not just as
    $R\to\infty$. Why does this have to be true for $G$ to be a valid Green's
    function of a *point* source?

    ??? success "Solution"
        The boundary integral is $\oint E(R)\,d\ell = E(R)\cdot 2\pi R = \frac{1}{2\pi R}\cdot 2\pi R = 1$,
        independent of $R$. It has to be
        radius-independent because a point source has no size: any circle
        that encloses the point encloses *all* of it, so the enclosed "charge"
        (and hence the flux, by the divergence theorem) cannot depend on how
        big a circle you drew. This is the same radius-independence argument
        behind [Ch. 19](19-einstein-radius.md#the-mean-convergence-identity)'s
        claim that $\bar\kappa(\theta_{\mathrm E})=1$ is a *definition*, not a
        coincidence: it is a statement about enclosed mass, and enclosed mass
        inside a fixed boundary doesn't care about the profile's shape outside
        it.

??? question "Exercise 6.3 — Derive $\kappa_{\mathrm{SIS}}$ by hand, in polar form"
    The SIS potential is $\psi_{\mathrm{SIS}}(\theta) = \theta_{\mathrm E}\,\theta$
    (radial, $\theta = |\boldsymbol\theta|$). (a) Compute
    $\nabla\psi_{\mathrm{SIS}}$ and confirm it matches
    `L.sis_deflection`'s constant-modulus form. (b) The 2-D Laplacian of a
    purely radial function $f(\theta)$ is $f''(\theta) + f'(\theta)/\theta$.
    Use this to get $\nabla^2\psi_{\mathrm{SIS}}$, then $\kappa_{\mathrm{SIS}}
    = \tfrac12\nabla^2\psi_{\mathrm{SIS}}$, and check it against the
    $0.3333$ this chapter computed numerically at $\theta=1.5''$.

    ??? success "Solution"
        (a) $\partial\psi/\partial\theta = \theta_{\mathrm E}$, and in 2-D
        $\nabla f(\theta) = f'(\theta)\,\hat\theta$, so
        $\nabla\psi_{\mathrm{SIS}} = \theta_{\mathrm E}\,\hat\theta$ —
        constant modulus $\theta_{\mathrm E}$, independent of $\theta$,
        exactly `L.sis_deflection`'s "flat rotation curve" behavior. (b)
        $f(\theta)=\theta_{\mathrm E}\theta$ gives $f'=\theta_{\mathrm E}$,
        $f''=0$, so
        $\nabla^2\psi_{\mathrm{SIS}} = 0 + \theta_{\mathrm E}/\theta = \theta_{\mathrm E}/\theta$.
        Then $\kappa_{\mathrm{SIS}}(\theta) = \theta_{\mathrm E}/(2\theta)$.
        At $\theta=1.5''$, $\theta_{\mathrm E}=1''$: $\kappa = 1/3 \approx 0.3333$ —
        the same number the finite-difference check landed on, this time from
        four lines of calculus instead of four function evaluations.

??? question "Exercise 6.4 — Why a proposed deflection field can be *wrong* even if it looks reasonable"
    Suppose someone hands you a candidate deflection field
    $\boldsymbol\alpha(\theta_1,\theta_2) = (\theta_2, -\theta_1)$ — smooth,
    well-defined everywhere, finite. Using the curl-free identity from [Divergence and the
    Laplacian](#divergence-laplacian), determine whether any lensing potential
    $\psi$ could produce it. What does this rule out about which vector
    fields are legitimate deflection fields at all?

    ??? success "Solution"
        Curl in 2-D is $\partial\alpha_2/\partial\theta_1 - \partial\alpha_1/\partial\theta_2$.
        Here that's
        $\partial(-\theta_1)/\partial\theta_1 - \partial\theta_2/\partial\theta_2 = -1 - 1 = -2 \neq 0$.
        Since $\nabla\times(\nabla\psi)\equiv 0$ for any smooth $\psi$, no
        potential can produce this field — it is a pure rotation (it sends
        every point to a 90-degree-rotated version of itself), and lensing
        deflections, being gradients of a scalar, can never rotate anything.
        This is a genuine, checkable constraint: any closed-form deflection a
        mass profile proposes (gigalens' EPL included) must satisfy
        $\partial\alpha_{\theta_1}/\partial\theta_2 =
        \partial\alpha_{\theta_2}/\partial\theta_1$ identically, or it does
        not correspond to any mass distribution at all.
