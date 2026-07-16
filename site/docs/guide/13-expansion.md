# 13. The expanding universe and redshift

This chapter answers a question every redshift number in this repository quietly
assumes you can already read: what does $z$ mean? Cikota's Einstein cross, the
Carousel cluster's background sheet, Ch. 12's DESI fiber-pair search — every one
of them reports $z$ as if its meaning were obvious. By the end of this chapter
you can say exactly what it asserts about the universe itself, derive Hubble's
law from a single Taylor expansion — the same move [Ch. 02](02-derivatives.md#taylor)
promised would recur through the whole book — and explain precisely why
$H_0 = 70$ is not, on its own, an age or a speed limit.

!!! abstract "What you can skip"
    Of the whole book, this chapter and the two after it (Ch. 14, Ch. 15) are
    the most detachable. The lens-modelling likelihood this repository actually
    optimises — the one Ch. 22 through Ch. 26 build up in full — is entirely
    angular: arcseconds in, arcseconds out, with no cosmological quantity
    anywhere in its gradient (`site/guide_src/cosmo.py:4`). Cosmology enters
    this repository's campaigns only where a physical, non-angular scale is
    required, and the money-number pipeline is not one of those places — see
    "Connect to the repo" below for the full, short list. If your only goal is
    $\gamma = 1.103$
    <!-- check: ch25.gamma_money = 1.103 ± 0.008 -->, skip straight to
    [Ch. 16](16-deflection.md#newtonian-deflection). Read this chapter instead
    before your first meeting with an astronomer: "the universe is expanding"
    is the one sentence every one of them assumes you already hold for the
    right reasons, and after this chapter, you will.

## The scale factor { #scale-factor }

Cosmologists describe the universe's size with a single dimensionless function
of time, the **scale factor** $a(t)$, normalized so that $a(t_0) \equiv 1$
**today**, at $t_0$. It carries no units and no meaning in isolation — only the
*ratio* of $a$ at two different times means anything, and that ratio is exactly
what a measured redshift hands you, in the next section.

The picture underneath $a(t)$: lay a fixed grid over the universe, with
galaxies (on average, ignoring their small individual peculiar motions) sitting
at fixed grid coordinates for all time — their **comoving** positions,
$\mathbf{x}_{\mathrm{c}}$. Nothing on this grid moves. What changes is the
ruler: the *physical* (proper) distance between two grid points at time $t$ is

$$
d_{\mathrm{phys}}(t) \;=\; a(t)\, d_{\mathrm{c}},
$$

where $d_{\mathrm{c}}$ is the fixed comoving separation between them. At
$t = t_0$, $a = 1$ and physical distance equals comoving distance — exactly
why the normalization was chosen that way.

!!! tip "You already know this"
    A comoving coordinate is a fixed label — like a vertex stored in
    normalized device coordinates, unchanged by anything downstream. The scale
    factor $a(t)$ is the single global multiplier that turns those fixed
    labels into physical distances at time $t$, the same way a viewport
    transform turns NDC into pixels. The mesh — every galaxy's comoving
    position — never moves. Only the multiplier changes, uniformly,
    everywhere, at once.

$a(t)$'s actual functional form — whether it grows like $t^{2/3}$, accelerates,
or does something else entirely — is set by how much matter and dark energy
the universe holds, and deriving it is [Ch. 14](14-frw.md#friedmann)'s job.
This chapter needs only $a(t)$'s value today and its instantaneous rate of
change right now — and that rate of change is exactly what "$H_0 = 70$"
reports.

## Hubble's law { #hubbles-law }

Define the **Hubble parameter** as the log-derivative of the scale factor,

$$
H(t) \;\equiv\; \frac{\dot a(t)}{a(t)} \;=\; \frac{d}{dt}\ln a(t).
$$

This is exactly the quantity you would call a continuously-compounded growth
rate in any other setting: the fractional change in size per unit time,
evaluated instantaneously. $a(t)$ need not actually grow exponentially —
$H(t)$ is a snapshot of its rate at $t$, not a claim about its long-run shape.

Taylor-expand $a(t)$ to first order around today, $t_0$, exactly as
[Ch. 02](02-derivatives.md#taylor) does for every other hard object in this
book:

$$
a(t) \;\approx\; 1 \;+\; H_0\,(t - t_0), \qquad H_0 \equiv H(t_0),
\label{eq:a-taylor}
$$

using $a(t_0) = 1$. This is nothing but a derivative given a name: the slope
of the line tangent to $a(t)$ at $t_0$.

Now put a comoving galaxy at fixed comoving distance $d_{\mathrm{c}}$. Its
proper distance at any nearby time is $d(t) = a(t)\, d_{\mathrm{c}}$, so its
recession velocity is

$$
v(t) \;=\; \frac{d}{dt}\big[a(t)\, d_{\mathrm{c}}\big]
\;=\; \dot a(t)\, d_{\mathrm{c}}
\;=\; \frac{\dot a(t)}{a(t)}\,\big[a(t)\, d_{\mathrm{c}}\big]
\;=\; H(t)\, d(t).
$$

Evaluated today, this is **Hubble's law**, $v = H_0\, d$. It is not an
empirical fit stapled onto the expansion after the fact; it is
$\eqref{eq:a-taylor}$'s linear term, restated as a velocity instead of a scale
factor.

Everywhere this repository touches cosmology, it fixes $H_0 = 70$ km/s/Mpc
(`site/guide_src/cosmo.py:29`). <!-- check: ch13.H0_km_s_mpc = 70.0 ± 0.1 -->
What else the model carries — $\Omega_m$, dark energy — is
[Ch. 14](14-frw.md#the-density-parameters)'s subject; only $H_0$ matters here.
The units are the tell: velocity over distance is $1/\text{time}$, a rate, not
a speed and not a length. Two readings of that rate:

- **Hubble time**, $1/H_0 = 13.97$ Gyr — roughly the age of the universe,
  though not exactly; [Ch. 14](14-frw.md#friedmann) derives why the two differ
  once matter and dark energy get a vote in $a(t)$'s history.
  <!-- check: ch13.hubble_time_gyr = 13.97 ± 0.01 -->
- **Hubble distance**, $c/H_0 = 4283$ Mpc — the characteristic length scale of
  the observable universe. <!-- check: ch13.hubble_distance_mpc = 4283 ± 1 -->

Equivalently, $H_0$ itself is $0.0716$ per Gyr: at today's rate, every
comoving distance grows about seven percent larger per billion years — the
same $d\ln a/dt$ from above, in units a human timescale can hold.
<!-- check: ch13.h0_per_gyr = 0.0716 ± 0.0005 -->

## Redshift as expansion { #redshift-is-expansion }

A photon is a wave, and as it crosses expanding space its wavelength is
carried along by the same scale factor that carries a comoving distance:

$$
\frac{\lambda_{\mathrm{obs}}}{\lambda_{\mathrm{emit}}}
\;=\; \frac{a(t_{\mathrm{obs}})}{a(t_{\mathrm{emit}})}
\;=\; \frac{1}{a(t_{\mathrm{emit}})},
$$

using $a(t_{\mathrm{obs}}) = a(t_0) \equiv 1$, since "observed" means "now."
[Ch. 12](12-spectroscopy.md#measuring-redshift) already showed you how $z$
gets measured — as exactly this wavelength ratio, off a spectral line whose
*rest* wavelength is known from the lab. Define it the same way here:

$$
1 + z \;\equiv\; \frac{\lambda_{\mathrm{obs}}}{\lambda_{\mathrm{emit}}}
\;=\; \frac{1}{a(t_{\mathrm{emit}})},
\qquad
a(t_{\mathrm{emit}}) = \frac{1}{1+z}.
$$

That is the whole content of "redshift is expansion": a measured $z$ is a
direct readout of how large the universe was when the light in front of you
left its source, relative to how large it is now. Not a velocity through
space — a size, then, compared against a size, now.

Four redshifts already sitting in this repository's worked examples translate
directly:

| system | $z$ | $a(t_{\mathrm{emit}}) = 1/(1+z)$ |
|---|---|---|
| Cikota+2023 Einstein-cross lens | 0.271 | 0.787 |
| Cikota+2023 Einstein-cross source | 0.897 | 0.527 |
| Carousel cluster lens ([Ch. 15](15-distances.md#the-carousel)) | 0.49 | 0.671 |
| Carousel reference plane | 1.432 | 0.411 |

<!-- check: ch13.z_cikota_lens = 0.271 ± 0.001 -->
<!-- check: ch13.scale_factor_cikota_lens = 0.787 ± 0.001 -->
<!-- check: ch13.z_cikota_source = 0.897 ± 0.001 -->
<!-- check: ch13.scale_factor_cikota_source = 0.527 ± 0.001 -->
<!-- check: ch13.z_carousel_lens = 0.49 ± 0.001 -->
<!-- check: ch13.scale_factor_carousel_lens = 0.671 ± 0.001 -->
<!-- check: ch13.z_carousel_ref = 1.432 ± 0.001 -->
<!-- check: ch13.scale_factor_carousel_ref = 0.411 ± 0.001 -->

The last row is the sharpest teaching example. It is tempting to read $z$ as a
velocity via $v \approx cz$ — and at low $z$ that is a good approximation,
since it is exactly $\eqref{eq:a-taylor}$'s linear term again, this time along
the photon's path instead of a nearby galaxy's worldline. Push it past where
the linear approximation holds and it breaks outright: read the Carousel's own
$z_{\mathrm{ref}} = 1.432$ as "$cz$" and you get 429,303 km/s, i.e.
$1.432\,c$ — faster than light.
<!-- check: ch13.naive_cz_carousel_kms = 429303 ± 50 -->
Nothing travels that fast, and nothing has to: no object is moving *through*
space at all, space itself is stretching, and there is no speed limit on how
fast a purely geometric distance can grow — the same way there is no upper
bound on how fast a resized window can grow even though nothing drawn inside
it is "moving." The correct statement was never a velocity. It is
$a(t_{\mathrm{emit}}) = 0.411$: the universe was 41 percent its current size
when that light left.

## Connect to the repo { #connect }

- `site/guide_src/cosmo.py:29` fixes the repo-wide choice,
  `COSMO = FlatLambdaCDM(H0=70, Om0=0.3)` — every number in this chapter that
  uses $H_0$ uses exactly this value.
- `site/guide_src/cosmo.py:4` states the audit this chapter's skip-box leans
  on: the lens-modelling likelihood is entirely angular, and cosmology enters
  campaigns in this repository only at a short, enumerated list of places:
    - `reproductions/hsu-2025/07_classify_einstein_dimple.py:50` — $\theta_{\mathrm{E}}$
      from $\sigma_v$, the calculation [Ch. 19](19-einstein-radius.md#theta-e-from-sigma-v)
      derives and Cikota's Einstein cross exercises.
    - `reproductions/sheu-2024b/04_setup_multiplane.py:35` — $\Sigma_{\mathrm{cr}}$
      and $M(<\theta_{\mathrm{E}})$ for the Carousel cluster, worked in full in
      [Ch. 15](15-distances.md#the-carousel).
    - `reproductions/sheu-2023/05_lightcurve_salt3.py:57` — a lensed
      supernova's Hubble-residual distance modulus, the one place in this
      repository $H_0$ sets an actual physical distance rather than an angle.
- `reproductions/claude-giga-lens/cgl/euclid_io.py:54` — the money-number
  pipeline itself fixes $z_l = 0.5$, $z_s = 1.0$ as inert defaults, "only the
  angular Einstein radius is meaningful."
  <!-- check: ch13.euclid_default_z_lens = 0.5 ± 0.001 -->
  <!-- check: ch13.euclid_default_z_src = 1.0 ± 0.001 -->
  Nothing in that pipeline's gradient depends on either number — this chapter's
  skip-box claim, confirmed from the campaign that matters most.

## Exercises { #exercises }

??? question "Exercise 13.1 — Hubble's law, both directions"
    Starting from $\eqref{eq:a-taylor}$, derive $v = H_0 d$ for a nearby
    comoving galaxy (two lines of algebra suffice). Then use it to predict the
    recession velocity of a galaxy at $d = 100$ Mpc, and invert $v \approx cz$
    to get the redshift that recession velocity implies.

    ??? success "Solution"
        Differentiate $d(t) = a(t)\, d_{\mathrm{c}}$:
        $\dot d = \dot a\, d_{\mathrm{c}} = (\dot a/a)\,(a\, d_{\mathrm{c}}) = H(t)\, d(t)$.
        At $t = t_0$, $H(t_0) = H_0$ and $d(t_0) = d$, so $v = H_0 d$.

        At $d = 100$ Mpc: $v = 70 \times 100 = 7000$ km/s.
        <!-- check: ch13.hubble_law_v_at_100mpc_kms = 7000 ± 1 -->
        Inverting $v \approx cz$: $z \approx v/c \approx 0.0233$.
        <!-- check: ch13.redshift_approx_at_100mpc = 0.0233 ± 0.0005 -->
        This is the same linear regime the naive-$cz$ trap in "Redshift as
        expansion" breaks out of once $z$ stops being small.

??? question "Exercise 13.2 — Two scale factors you will need again"
    Compute $a(t_{\mathrm{emit}}) = 1/(1+z)$ for $z = 0.5$ and $z = 2.0$.
    [Ch. 15](15-distances.md#distances-do-not-add) uses exactly this pair to
    show that angular-diameter distances do not subtract. State in one
    sentence what each scale factor means physically.

    ??? success "Solution"
        $a(0.5) = 1/1.5 = 0.667$.
        <!-- check: ch13.scale_factor_z05 = 0.667 ± 0.001 -->
        $a(2.0) = 1/3 = 0.333$.
        <!-- check: ch13.scale_factor_z20 = 0.333 ± 0.001 -->
        The $z=0.5$ light left when the universe was two-thirds its current
        size; the $z=2.0$ light left when it was one-third — a factor of two
        in redshift is not a factor of two in "how far back," because $a$ and
        $z$ are related by $1/(1+z)$, not linearly.

??? question "Exercise 13.3 — Why $c/H_0$ has units of length"
    $H_0$ is quoted in km/s/Mpc. Show, by unit algebra alone, that $c/H_0$ has
    units of length, then compute it and compare it to the Hubble distance
    quoted above.

    ??? success "Solution"
        $[c]/[H_0] = (\text{km/s}) / (\text{km s}^{-1}\,\text{Mpc}^{-1}) = \text{Mpc}$
        — the km/s cancels completely, leaving a pure length. Numerically,
        $c/H_0 = 4283$ Mpc, matching the Hubble distance quoted in "Hubble's
        law" above. <!-- check: ch13.hubble_distance_mpc = 4283 ± 1 -->
        Run the same cancellation on $1/H_0$ instead and what is left is a
        pure time — the Hubble time.

??? question "Exercise 13.4 — The naive-$cz$ trap, generalised"
    The Carousel's reference redshift, read as $cz$, gave $1.432\,c$ — faster
    than light. At roughly what redshift does the naive Newtonian reading
    $v = cz$ first predict a superluminal recession, and why does crossing
    that threshold not violate special relativity?

    ??? success "Solution"
        $v = cz$ exceeds $c$ as soon as $z > 1$ — no computation needed beyond
        noticing $cz > c \iff z > 1$. Both the Carousel's reference plane
        ($z_{\mathrm{ref}} = 1.432$) and the naive-$cz$ speed derived from it
        (429,303 km/s) sit past that threshold.
        <!-- check: ch13.z_carousel_ref = 1.432 ± 0.001 -->
        <!-- check: ch13.naive_cz_carousel_kms = 429303 ± 50 -->
        Special relativity bounds the speed of anything moving *through*
        space; it says nothing about how fast a purely geometric distance
        between two points can grow while nothing moves through the space in
        between. $v = cz$ is only ever the low-$z$ approximation to
        $a(t_{\mathrm{emit}}) = 1/(1+z)$ — the actual, always-valid statement
        — so its failure at high $z$ is the approximation's fault, not
        relativity's.
