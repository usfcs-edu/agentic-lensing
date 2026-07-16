# The correlated-noise likelihood: whitening a real telescope

[Ch. 11](11-observation.md#why-drizzle-correlates-noise) told you drizzle
correlates pixel noise and a diagonal likelihood assumes it does not; [Ch.
7](07-fourier.md#whitening) built the Fourier machinery — a stationary kernel,
a whitening filter, an operator-norm gate — that a fix would need. This
chapter is where those two pieces become one executable object: $C =
D^{1/2}KD^{1/2}$, fit on a real, masked, model-subtracted residual, applied by
a local convolution, and tested rigorously enough that a real implementation
defect got caught before it ever touched a published number. By the end you
can read `reproductions/claude-giga-lens/papers/main.tex`'s entire Methods I
section (`cgl/whiten.py` and `cgl/noise.py` end to end) and know exactly which
claim each gate backs. This chapter does not defend $\gamma=1.103$ — that
verdict is [Ch. 25](25-money-number.md#the-chain)'s job. It builds, and
proves correct, the machine that number comes out of.

!!! abstract "What you can skip"
    Nothing here re-derives the whitening filter itself
    ($\hat h[k]=1/\sqrt{S[k]}$, the operator-norm gate $e_{\mathrm{op}}$, the
    spectral floor) — that is [Ch. 7](07-fourier.md#whitening), assumed in
    full. Nothing here re-derives the noise model $D$ or the drizzle mechanism
    — that is [Ch. 11](11-observation.md#the-noise-model), also assumed. What
    is new: how the correlation kernel $K$ actually gets estimated from a
    masked real image, the full acceptance table across all three real
    products plus one pre-registered exception, the rule that decides which
    pixels survive being whitened by a local filter, and — the chapter's
    center of gravity — why the whole apparatus must reduce *exactly* to the
    diagonal case, both as an algebraic identity and as the regression test
    that caught a genuine compiler defect.

## The covariance model { #the-covariance-model }

[Ch. 11](11-observation.md#the-noise-model) built $D=\mathrm{diag}(\sigma_i^2)$,
the per-pixel heteroscedastic variance (masked pixels inflated to
$\sigma\to10^{10}$); [Ch. 7](07-fourier.md#psd-and-autocorrelation) built the
idea of a stationary correlation matrix diagonalized by the Fourier basis.
This chapter's whole job is the object that glues the two into one pixel
covariance,

$$
C \;=\; D^{1/2}\,K\,D^{1/2},
\label{eq:cov}
$$

`reproductions/claude-giga-lens/papers/main.tex:367`
([Methods I, covariance model and drizzle anchor](../current/claude-giga-lens/index.md#sec:methods1)).
(This $D$ is unrelated to [Ch. 15](15-distances.md#three-distances)'s angular-diameter
distances $D_{\mathrm{d}}, D_{\mathrm{s}}, D_{\mathrm{ds}}$ — same repo, same
letter, a different quantity every time context makes it obvious which.)
$D^{1/2}$ carries every pixel's noise *level*; $K$ — a pure correlation
matrix, diagonal entries exactly $1$ — carries the *shape* of how
neighbouring pixels' noise moves together, independent of how bright any of
them are. Factoring the two apart is what makes $K$ estimable at all: one
stationary kernel $\rho(\Delta)$, the same at every pixel, stands in for what
would otherwise be an $N\times N$ covariance no cutout could ever constrain
directly.

Two design choices matter before any fitting starts. First, $K$ is fixed
*per product* — estimated once, before sampling, never re-estimated as the
mass model $\theta$ moves, which is why $-\tfrac12\log\det C$ never has to be
recomputed inside the sampler's inner loop ([The diagonal
limit](#the-diagonal-limit) cashes this out). Second, $K$ must be fit on the
*model-subtracted* residual, never the raw image — the identical discipline
`reproductions/claude-giga-lens/cgl/guards.py:129`
(`assert_model_subtracted_sky`) already enforced in
[Ch. 11](11-observation.md#the-noise-model) for $\sigma_{\mathrm{bkg}}$,
now doing double duty: fit a "noise" correlation on an image that still
contains the lens galaxy's own smooth light, and you are measuring the
autocorrelation of a *galaxy*, not of noise.

Estimating $\rho(\Delta)$ from a real cutout means estimating it from a
*masked* one — bad pixels, saturated cores, the segmentation mask itself all
remove pixels from play, and a naive autocorrelation over what remains is
biased, because the number of valid pairs at each lag $\Delta$ is not
constant. `masked_autocorr_full`
(`reproductions/claude-giga-lens/cgl/noise.py:47`) fixes this with the
identical Wiener–Khinchin move [Ch. 7](07-fourier.md#psd-and-autocorrelation)
built, applied *twice*: once to the mean-subtracted, masked residual $n$, and
once to the mask $w$ itself ($0$/$1$-valued),

$$
\mathrm{acf} = \mathrm{IFFT}\!\big(|\mathrm{FFT}(n)|^2\big), \qquad
\mathrm{wgt} = \mathrm{IFFT}\!\big(|\mathrm{FFT}(w)|^2\big), \qquad
\rho[\Delta] = \frac{(\mathrm{acf}/\mathrm{wgt})[\Delta]}{(\mathrm{acf}/\mathrm{wgt})[0]},
$$

where $\mathrm{wgt}[\Delta]$ is exactly the number of valid pixel *pairs*
separated by $\Delta$ — the mask's own autocorrelation, computed by the same
padded-FFT trick used on the data, deconvolves the mask's footprint out of
the estimate. Lags where fewer than $5\%$ of the maximum pair count survive
($\mathrm{wgt}[\Delta] < 0.05\,\mathrm{wgt}[0]$) are marked invalid rather
than trusted.

## Fitting the kernel { #fitting-the-kernel }

The pre-registered plan for $K$ was the simplest family that could plausibly
work — one Gaussian smoothing of the drizzle-anchored correlation blended
with a delta, $\rho=(1-w)\delta + w[\rho_{\mathrm{drz}}\circledast
G(\sigma_e)]$ — and [Ch. 7](07-fourier.md#positive-semidefiniteness) already
told you it fails: on every one of the three real drizzle products, its best
achievable fit misses the campaign's own $\max|\rho_{\mathrm{fit}}-
\rho_{\mathrm{meas}}|\le0.05$ gate, by a wide margin.

| product | single-Gaussian, worst resid. | two-component, worst resid. | gate |
|---|---|---|---|
| `v2d` (native) | <!-- check: ch24.kernel_single_family_v2d = 0.0895 ± 0.0005 --> $0.0895$ FAIL | <!-- check: ch24.kernel_two_component_v2d = 0.0448 ± 0.0005 --> $0.0448$ PASS | $\le0.05$ |
| `v3` (fine)    | <!-- check: ch24.kernel_single_family_v3 = 0.3110 ± 0.0005 --> $0.3110$ FAIL | <!-- check: ch24.kernel_two_component_v3 = 0.0270 ± 0.0005 --> $0.0270$ PASS | $\le0.05$ |
| `v3b` (binned) | <!-- check: ch24.kernel_single_family_v3b = 0.1664 ± 0.0005 --> $0.1664$ FAIL | <!-- check: ch24.kernel_two_component_v3b = 0.0326 ± 0.0005 --> $0.0326$ PASS | $\le0.05$ |

(`reproductions/claude-giga-lens/data/noise_kernel_report.json`, fields
`products.*.fit_single_family.max_abs_resid` and
`products.*.max_abs_resid`; `reproductions/claude-giga-lens/papers/main.tex:384`–`397`.) These residuals were
not tuned to fail: the single-Gaussian family was the *first* thing tried,
and every real product's model-subtracted residual carries structure it
cannot represent — an anisotropic core (the drizzle kernel's own $3\times3$
tent is not circular), a medium-scale correlated pedestal the $1''$
background detrend leaves behind, and, on the native product specifically,
column stripes. The two-component family the campaign adopted instead adds
exactly one more term, a second, broader bivariate-Gaussian pedestal
independent of the drizzle-anchored core,

$$
\rho \;=\; (1-w_d-w_b)\,\delta
\;+\; w_d\big[\rho_{\mathrm{drz}}\circledast G_2(\sigma_{ey},\sigma_{ex},c_e)\big]
\;+\; w_b\,G_2(\sigma_{by},\sigma_{bx},c_b),
\label{eq:kernelfam}
$$

(`reproductions/claude-giga-lens/cgl/noise.py:354`, `rho_model2`;
`reproductions/claude-giga-lens/papers/main.tex:389`–`394`). Each piece is individually positive
semi-definite — [Ch. 7](07-fourier.md#positive-semidefiniteness) proves why: a
delta has flat spectrum $S\equiv1$, a Gaussian's Fourier transform is itself a
positive Gaussian, and convolution multiplies spectra — so any non-negative
combination with $w_d+w_b\le1$ is too, a constraint you can check directly on
the fitted "money" product `v3b`: $w_d=$
<!-- check: ch24.w_d_v3b = 0.7333 ± 0.001 --> $0.733$, $w_b=$
<!-- check: ch24.w_b_v3b = 0.2483 ± 0.001 --> $0.248$, summing to
<!-- check: ch24.w_sum_v3b = 0.9816 ± 0.001 --> $0.982\le1$.

!!! tip "You already know this"
    "A non-negative sum of PSD kernels is PSD" is not a fact borrowed from
    astronomy — it is the closure property that lets an SVM combine a
    polynomial and an RBF kernel into one still-valid Mercer kernel. This
    family is built the identical way: three individually-valid pieces,
    summed with non-negative weights, so the whole is guaranteed valid
    without ever checking an eigenvalue directly.

An independent cross-scale check adds confidence the fit is not chasing
product-specific noise: `binned_kernel_from_fine`
(`reproductions/claude-giga-lens/cgl/noise.py:481`) exactly block-sums the
*fitted fine* kernel down to binned resolution and compares it against the
ACF measured directly on the block-summed fine residual. The two agree to
<!-- check: ch24.blocksum_crosscheck_diff = 0.0308 ± 0.0005 --> $0.031$
against the same $0.05$ gate — confirming the kernels commute correctly with
$2\times2$ binning, a purely mathematical property unrelated to how well any
one kernel happens to fit.

## Convolutional whitening { #convolutional-whitening }

[Ch. 7](07-fourier.md#whitening) derived the whitening filter itself,
$\hat h[k]=1/\sqrt{S[k]}$, truncated to a local $(2M+1)^2$ stencil and
accepted only if its operator-norm gate
$e_{\mathrm{op}}=\max_\omega|S(\omega)|\hat h_M(\omega)|^2-1|\le0.02$ holds.
This section is the operational residue of that derivation: the whiteners the
campaign actually accepted, on the three real products, plus the rule that
decides which pixels survive being whitened by a *local* filter at all.

| product | $M$ | $e_{\mathrm{op}}$ | gate | kept / eroded |
|---|---|---|---|---|
| `v2d` (native, strict) | $14$ | <!-- check: ch24.eop_v2d = 0.0177 ± 0.0005 --> $0.0177$ | $\le0.02$ | $5{,}865\,/\,487$ |
| `v3` (fine)            | $20$ | <!-- check: ch24.eop_v3 = 0.0160 ± 0.0005 --> $0.0160$ | $\le0.02$ | $66{,}752\,/\,37{,}519$ |
| `v3b` (binned)         | $10$ | <!-- check: ch24.eop_v3b = 0.0124 ± 0.0005 --> $0.0124$ | $\le0.02$ | $16{,}653\,/\,9{,}273$ |
| `v2d_relaxed` (native) | $10$ | <!-- check: ch24.eop_v2d_relaxed = 0.0312 ± 0.0005 --> $0.0312$ | $\le0.05^{\ast}$ | $5{,}865\,/\,1{,}466$ |

(`reproductions/claude-giga-lens/data/whitener_report.json`;
`reproductions/claude-giga-lens/papers/main.tex`, Table `tab:whiten`.) $^{\ast}$See below. Every stencil is
a *local* filter — a pixel's whitened value depends on the $(2M+1)^2$ pixels
around it — so a masked neighbour, or one off the image edge (the
`'SAME'`-padded convolution zero-pads borders, and zero is not data),
contaminates that one output. `erode_keep`
(`reproductions/claude-giga-lens/cgl/whiten.py:274`) handles this by
*dropping*, never down-weighting: a pixel survives into the whitened vector
only if its entire stencil is clean, computed by `scipy.ndimage.binary_erosion`
on the keep-mask with the $(2M+1)^2$ stencil as structuring element,
intersected with an explicit $M$-pixel interior border trim.

!!! tip "You already know this"
    This is exactly the difference between `'same'` and `'valid'` padding in
    a convolutional layer, applied for the same reason: `'same'` padding
    manufactures zeros at the border and lets the network pretend they are
    real input; `'valid'` padding instead *shrinks* the output, keeping only
    positions whose full receptive field saw genuine data. `erode_keep` is
    `'valid'` padding for a statistical estimator, where "genuine data" means
    "not masked."

The rule is statistically exact, but it costs, and the native product pays
hardest: the strict `v2d` whitener's $29^2$ stencil ($M=14$) erodes away
<!-- check: ch24.pixel_loss_frac_v2d_strict = 0.917 ± 0.001 --> $91.7\%$ of
the product's pixels — $5{,}865$ down to just $487$ kept. That is not a
construction defect; it is what a wide stencil genuinely costs against a
mask with any interior structure. A pre-registered, evidence-driven exception
widens the *acceptance gate itself* from $0.02$ to $0.05$ — not merely
accepting a worse fit under the old bar — for a `v2d_relaxed` whitener at a
smaller $M=10$, recovering
<!-- check: ch24.n_eroded_vs_strict_v2d = 3.01 ± 0.01 --> $3.0\times$ as many
kept pixels ($1{,}466$), adopted only after mock recovery confirmed it still
calibrates correctly (`reproductions/claude-giga-lens/CAMPAIGN.md`, P1b
diagnosis, 2026-07-08).

## The diagonal limit { #the-diagonal-limit }

Every piece built so far exists to answer one question honestly: given a
real, non-trivial $K$, does the correlated likelihood compute what it claims
to? The cheapest, most decisive way to ask that is to make $K$ *trivial* on
purpose and check that the whole apparatus degenerates back to something
already validated. `make_conv_whitener`
(`reproductions/claude-giga-lens/cgl/whiten.py:39`) computes

$$
u \;=\; \mathrm{keep}_w \odot \big(h * (D^{-1/2}\,\mathrm{image})\big),
$$

where $*$ is a `'SAME'`-padded 2-D convolution. Set $h=[[1.0]]$ — a single
tap, the $1\times1$ "delta" kernel: it has no neighbour to sum in, so a
`'SAME'`-padded correlation with a $1\times1$ kernel *is* elementwise
multiplication by that one number,

$$
u \;=\; \mathrm{keep}_w \odot \big(D^{-1/2}\,\mathrm{image}\big).
\label{eq:gateD}
$$

With $D^{-1/2}=1/\text{masked\_err\_map}$ and $\mathrm{keep}_w=\text{keep\_mask}$,
the right-hand side of $\eqref{eq:gateD}$ is not *similar* to
`cgl/likelihood.py`'s original diagonal path (`R * sqrtW`,
`reproductions/claude-giga-lens/cgl/likelihood.py:358`) — it *is* that path,
term for term. `cgl/whiten.py:20` names this explicitly: "Parity anchor
(gate D)."

**Worked example.** You can rebuild the identity yourself in three lines of
plain NumPy, no `jax` required: a small noisy image, its own per-pixel
$\sigma$, a mask with one bad pixel.

```python
sqrt_d_inv = 1.0 / sigma
h = np.array([[1.0]])                            # gate D's delta kernel
u_conv = keep * (h[0, 0] * sqrt_d_inv * img)      # make_conv_whitener, h=[[1]]
u_diag = keep * (img * sqrt_d_inv)                # the diagonal path
```

The two agree to
<!-- check: ch24.toy_delta_kernel_matches_diagonal = 0.0 ± 1e-12 --> exactly
$0$ — not "close," because $\eqref{eq:gateD}$ has no approximation in it,
only algebra. The production harness runs the identical check on the real
$46$-parameter marginalized posterior, at four perturbed reference points,
and reports the same answer:
<!-- check: ch24.gate_D_achieved = 0.0 ± 1e-12 --> $|\Delta\log p|=0$ against
a gate of <!-- check: ch24.gate_D_threshold = 1e-10 ± 1e-14 --> $10^{-10}$
(`reproductions/claude-giga-lens/data/parity_report.json`, `gates.D`).

!!! tip "You already know this"
    Gate D is a golden-file test — the kind you write when a refactor is
    supposed to be a mathematical no-op: assert bit-for-bit agreement against
    the pre-refactor path, not "looks about right." A tolerance here would be
    a category error. If $h=[[1]]$ ever stopped reproducing the diagonal
    path to machine precision, the conv-whitening code would have a bug, full
    stop — there is no numerical approximation between the two expressions to
    excuse a gap.

An exact identity test earns its keep the day it fails. Gate D is not the
reason the campaign found its worst implementation defect, but it sits right
next to the reason. Whitening the $28$ shapelet design columns with
`jax.vmap(whiten_fn, in_axes=2)`
(`reproductions/claude-giga-lens/cgl/likelihood.py:367`) — $28$ separate
convolution ops, vmapped — livelocked the XLA compiler under the
reverse-mode autodiff a sampler needs for its gradient: $100\%$ CPU, GPU
idle, more than $20$ minutes with no output, versus $14$ seconds for the
diagonal path (no convolution at all). `_grouped_whiten_ops`
(`reproductions/claude-giga-lens/cgl/e2.py:218`) collapses the $28$
per-column convolutions into *one* depthwise (`feature_group_count`)
convolution — the same $28$ filters, structured so XLA can fuse them as a
single op — and the batched gradient compiles in
<!-- check: ch24.grouped_conv_compile_s = 13.8 ± 0.05 --> $13.8\,\mathrm{s}$,
log-posterior asserted bit-identical to the original at build time, costing
at most <!-- check: ch24.conv_fwd_over_diag_ratio = 1.6 ± 0.01 --> $1.6\times$
the diagonal forward pass and
<!-- check: ch24.conv_grad_over_diag_ratio = 1.545 ± 0.01 --> $1.55\times$
its gradient ($40$/$85$ vs $25$/$55\,\mathrm{ms}$ on an L4) — diagonal-comparable
throughput, bought only because a defect that had nothing to do with the math
being *wrong* got caught by testing the math's *structure*, not its output.

A second, genuinely different check completes the validation. Gate D
certifies the whitened functional $C^{-1}:=G_e^\top G_e$ against a
*degenerate* special case ($K=I$) where the right answer is already known
by construction; it says nothing about a real, non-trivial $K$. For that,
the campaign compares the convolution-whitened $\log L$ against an
independent CPU float-64 *dense*-covariance Cholesky solve on the kept
pixels, at $20$ random prior draws: worst disagreement
<!-- check: ch24.dense_c_worst_dlogl_v2d = 2.79e-9 ± 1e-10 --> $2.79\times10^{-9}$
nat (`v2d`) and
<!-- check: ch24.dense_c_worst_dlogl_v3b = 6.26e-7 ± 1e-8 --> $6.26\times10^{-7}$
nat (`v3b`), both far inside a $0.1$-nat gate
(`reproductions/claude-giga-lens/data/exact_ref_report.json`). Gate D asks
"does this reduce to a case I already trust?"; the dense reference asks
"does it agree with an independently-coded computation of the *same real*
answer?" — passing only one would leave a genuine hole.

One more caution belongs here, precisely because an exact-identity test can
hide it. `cgl/marg.py`'s log-likelihood drops the additive constant
$-\tfrac12\log\det C$ — correct for *sampling*, since $K$ is fixed per
product and that term never moves as $\theta$ does (the point
[The covariance model](#the-covariance-model) opened with). But the *dropped*
constant differs across whiteners, because different $K$'s have different
determinants, so comparing evidences across whiteners without restoring it
silently compares apples whose stems were cut to different lengths. The
campaign's own exact-vs-Szegő gap — exact $\log\det C$ minus the per-pixel
Szegő-limit approximation
[Ch. 7](07-fourier.md#psd-and-autocorrelation) built — is
<!-- check: ch24.szego_gap_v2d = 27.30 ± 0.02 --> $+27.30$ nat on `v2d` and
<!-- check: ch24.szego_gap_v3b = 179.21 ± 0.02 --> $+179.21$ nat on `v3b`
(`reproductions/claude-giga-lens/data/exact_ref_report.json`,
`constants.szego_gap`) — both dwarfing the handful of nats that separate a
"decisive" Bayes factor from an ambiguous one. This is exactly why
[Ch. 25](25-money-number.md#the-evidence-flip)'s evidence comparison is done
*within* one fixed whitener, never across two.

Every gate in this chapter passing proves the machinery is correct — that
$C=D^{1/2}KD^{1/2}$ really is what gets evaluated, to machine precision where
an exact answer exists and to a fraction of a nat where only an independent
numerical reference does. It proves nothing yet about whether that correct
machinery, pointed at a real, near-singular, drizzle-correlated system,
returns a *trustworthy* $\gamma$. In the campaign's own words, the
correlated likelihood "proves necessary but not sufficient"
(`reproductions/claude-giga-lens/papers/main.tex:129`–`130`) — necessary,
because every gate here is the precondition for pointing the machinery at
real data at all; not sufficient, because
[Ch. 25](25-money-number.md#the-chain) still has to ask whether the number
that comes out the other end is one you should believe.

!!! note "γ Ledger"
    This chapter builds the machine, not the verdict. Nothing here computes a
    value of $\gamma$ — every gate is a statement about the *likelihood*,
    never about the mass model it will eventually be pointed at. What it
    rules in: the correlated likelihood that produces
    $\gamma_{\mathrm{binned}}(\mathrm{corr,\,low})=1.103$ is not a black box
    taken on faith — every moving part (the kernel fit, the whitener, the
    reduction to the diagonal case, the dense-covariance cross-check) is
    independently, exactly verified, and the one implementation defect this
    machinery could have hidden silently — the `vmap` livelock — was caught,
    not shipped. What it rules out: nothing about $1.103$ itself. Every gate
    here passing is the precondition for [Ch. 25](25-money-number.md#the-money-number)
    being allowed to trust the number this machinery returns; it is not, by
    itself, a reason to believe that number is right.

## Connect to the repo { #connect }

- [`cgl/whiten.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/whiten.py) —
  `make_conv_whitener` (line `39`, this chapter's diagonal-limit derivation),
  `build_whitener` (line `95`), `erode_keep` (line `274`), and the "Parity
  anchor (gate D)" docstring (line `20`).
- [`cgl/noise.py`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/noise.py) —
  `masked_autocorr_full` (line `47`, the mask-deconvolved ACF), `rho_model2`
  / `fit_kernel2` (lines `354` / `392`, the two-component kernel family),
  `binned_kernel_from_fine` (line `481`, the cross-scale check).
- [`cgl/guards.py:129`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/guards.py#L129)
  (`assert_model_subtracted_sky`) — the same guard [Ch. 11](11-observation.md#the-noise-model)
  introduced, now protecting $K$'s fit too.
- [`cgl/e2.py:218`](https://github.com/usfcs-edu/agentic-lensing/blob/main/reproductions/claude-giga-lens/cgl/e2.py#L218)
  (`_grouped_whiten_ops`, `build_marg_model_grouped`) — the grouped-convolution
  fix for the gate-D-adjacent livelock, asserted bit-identical to
  `cgl/likelihood.py`'s original at build time.
- `reproductions/claude-giga-lens/data/noise_kernel_report.json`,
  `data/whitener_report.json`, `data/exact_ref_report.json`,
  `data/parity_report.json` — this chapter's worked numbers, read directly by
  `site/guide_src/worked_examples.py`'s `ch24_correlated_noise`.
- [Methods I](../current/claude-giga-lens/index.md#sec:methods1),
  [Convolutional whitening](../current/claude-giga-lens/index.md#sec:whiten),
  and [Exact-reference validation and the grouped-convolution fix](../current/claude-giga-lens/index.md#sec:exactref) —
  the report prose this chapter exists to make readable.
- [Ch. 7](07-fourier.md#whitening) derives the whitening filter and the
  $e_{\mathrm{op}}$ gate this chapter only applies; [Ch. 11](11-observation.md#why-drizzle-correlates-noise)
  is where $D$, and the whole motivation for $K$, come from;
  [Ch. 22](22-inference.md#the-occam-term) is where `whiten_fn` plugs into the
  marginalized likelihood this chapter whitens; [Ch. 25](25-money-number.md#the-chain)
  is the verdict this machinery makes possible.

## Exercises { #exercises }

??? question "Exercise 24.1 — Derive the diagonal limit, don't just quote it"
    `make_conv_whitener(h, sqrt_d_inv, keep_w)` computes
    $u = \mathrm{keep}_w \odot \big(h * (\mathrm{sqrt\_d\_inv} \odot
    \mathrm{image})\big)$, where $*$ is a `'SAME'`-padded 2-D convolution.
    Set $h=[[c]]$ for an arbitrary single scalar $c$ (not necessarily $1$),
    and show algebraically that $u$ reduces to an elementwise product no
    matter what $c$ is. Then explain why $c=1$ specifically is required for
    $\eqref{eq:gateD}$ to match the *original* diagonal path exactly, rather
    than merely being proportional to it.

    ??? success "Solution"
        A $1\times1$ kernel has a single tap at lag $(0,0)$, and `'SAME'`
        padding does not change the output size, so at every output position
        the convolution sum has exactly one term: $h[0,0]\cdot v$ where
        $v=\mathrm{sqrt\_d\_inv}\odot\mathrm{image}$. There is no neighbour
        to sum in regardless of $c$, so $u = \mathrm{keep}_w \odot (c\,v)$
        for *any* scalar $c$ — the "convolution" is always elementwise
        multiplication by that one number, whatever it is. The original
        diagonal path is $R\cdot\sqrt W = R/\sigma$ exactly, with no extra
        factor, so matching it exactly (not merely up to a constant
        rescaling of the whole likelihood) requires $c=1$: any other $c$
        would still collapse to elementwise multiplication, but by the wrong
        number, and gate D would report a nonzero — in fact systematic, not
        numerical-noise-sized — $|\Delta\log p|$.

??? question "Exercise 24.2 — Two different tests, two different failure modes"
    This chapter ran gate D (the delta-kernel identity) and the
    dense-covariance Cholesky cross-check side by side. Construct a concrete
    bug in `make_conv_whitener` that gate D would *miss* but the
    dense-covariance reference would *catch*, and a second bug where the
    reverse is true.

    ??? success "Solution"
        A bug gate D misses: a sign error only in $h$'s off-center taps,
        center tap $h[M,M]$ still correct. At $h=[[1]]$ there *are* no
        off-center taps — the $1\times1$ case cannot exercise a bug that only
        exists in a $\ge3\times3$ kernel — so gate D reports exact agreement
        while every real whitener silently misapplies its neighbours; the
        dense-covariance reference, which uses a genuinely non-trivial $K$,
        catches it immediately as a nonzero $\Delta\log L$. A bug the dense
        reference might plausibly miss but gate D catches: `keep_w` built
        from the wrong mask array, one that happens to be all-ones on the
        $20$ prior draws the dense check samples — the two could then agree
        by chance on the case actually tested. Gate D's own construction,
        built with a deliberately masked pixel ([The diagonal
        limit](#the-diagonal-limit)'s worked example), is far more likely to
        expose it: an exact-equality test at an adversarial input has
        nowhere to hide an input-dependent error the way a "close enough"
        comparison does.

??? question "Exercise 24.3 — Two log-determinants, one likelihood, different lifespans"
    [Ch. 22](22-inference.md#the-occam-term) builds the Occam term
    $-\tfrac12\log\det A$, with $A=\tilde X^\top\tilde X+\Lambda$ (the ridge
    normal matrix `cgl/marg.py` builds — *not* [Ch. 5](05-linear-algebra.md#symmetric-2x2)'s
    lens Jacobian, also called $A$, which never appears in this chapter),
    computed fresh at every posterior evaluation. This chapter's
    $-\tfrac12\log\det C$ is dropped as a constant and never recomputed
    during sampling. Both are log-determinants inside the same
    log-likelihood expression. What property of the two matrices explains
    why one must be recomputed every step and the other can be dropped
    entirely?

    ??? success "Solution"
        $A=\tilde X^\top\tilde X+\Lambda$ depends on $\tilde X$, the whitened
        *design matrix* built from the lensed, PSF-convolved shapelet basis
        images — which depend on the mass-model parameters $\theta$ (the
        deflection field that lenses the source). Move $\theta$ and $\tilde
        X$ moves, so $A$'s determinant moves too: the Occam term is a genuine
        function of $\theta$, recomputed every evaluation. $C=D^{1/2}KD^{1/2}$
        depends only on the fixed error map $D$ and the fixed, once-fit
        kernel $K$ ([The covariance model](#the-covariance-model)) — neither
        depends on $\theta$ at all. A constant independent of the sampled
        parameter contributes nothing to the posterior's *shape* and can be
        dropped for sampling, which is why it is safe to drop
        $-\tfrac12\log\det C$ but never $-\tfrac12\log\det A$.

??? question "Exercise 24.4 — Why widen the gate rather than just accept a worse number"
    [Convolutional whitening](#convolutional-whitening) described
    `v2d_relaxed` as widening the *acceptance gate* from $0.02$ to $0.05$, as
    a pre-registered, dated exception — not simply using whichever whitener
    the strict $M=14$ construction happened to produce. Why does that
    distinction matter for whether the resulting posterior can be trusted?

    ??? success "Solution"
        A gate that gets ignored quietly whenever a result disappoints is
        not a gate; it is a suggestion, and it opens exactly the failure mode
        this book keeps returning to (the retracted $\chi^2_\nu=0.451$,
        [Ch. 11](11-observation.md#the-noise-model)): a number that looks
        good only because the bar that would have flagged it moved without
        anyone recording why. Widening the gate itself, dated and justified
        *before* the resulting whitener is used for any science
        (`reproductions/claude-giga-lens/CAMPAIGN.md`, P1b diagnosis,
        2026-07-08), and then separately validating that whitener against
        mocks with *known* truth, keeps the decision auditable: anyone can
        check whether $0.0312\le0.05$ was decided before or after seeing
        whether it helped the money number. Silently accepting whatever
        $e_{\mathrm{op}}$ a construction happens to produce would make every
        subsequent number unfalsifiable, because no fixed bar would ever
        actually have been crossed.
